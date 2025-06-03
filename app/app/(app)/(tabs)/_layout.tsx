import { Tabs } from 'expo-router'
import { EvilIcons } from '@expo/vector-icons';
import TabHeader from '../../_components/TabHeader';

const ICON_SIZE = 30;

interface optionsInterface {
    title: string,
    color: string,
    iconName: "comment" | "image" | "gear",
    addButton: boolean
}

const options = ({ title, color, iconName, addButton }: optionsInterface) => {
    return {
        header: () => <TabHeader title={title} addButton={addButton} backgroundColor={color} />,
        tabBarActiveTintColor: color,
        tabBarLabel: title,
        tabBarIcon: (({ color }: any) => <EvilIcons color={color} name={iconName} size={ICON_SIZE} />),
    }
}

export default function _layout() {
    return <Tabs screenOptions={
        {
            animation: 'shift',
            headerTitleAlign: 'left',
            tabBarStyle: { height: 60 },
            tabBarLabelPosition: 'below-icon',
        }
    }>
        <Tabs.Screen name='index' options={options({ title: 'chats', color: '#6ee7b7', iconName: 'comment', addButton: true, })} />

        <Tabs.Screen name='explore' options={options({ title: 'explore', color: '#a5b4fc', iconName: 'image', addButton: false, })} />

        <Tabs.Screen name='settings' options={options({ title: 'settings', color: '#0891b2', iconName: 'gear', addButton: false, })} />
    </Tabs>;
}


